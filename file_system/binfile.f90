module Readbin
    use, intrinsic :: iso_fortran_env
    implicit none
    private
    public bswap32, read_int32, read_images, read_label
contains
    function bswap32(n) result(r)
        integer(int32), intent(in) :: n
        integer(int32) :: r
        integer(int32) :: p1,p2,p3,p4
        p1 = iand(ishft(n, -24), 255)
        p2 = iand(ishft(n,-8),255)
        p3 = ishft(iand(n, 65280),8)
        p4 = ishft(iand(n, 255), 24)

        r = ior( ior( ior( p1, p2 ), p3), p4 )
    end function bswap32

    subroutine read_int32(unit, value)
        integer(int32), intent(in) :: unit
        integer(int32), intent(out) :: value

        read(unit) value
        value = bswap32(value)
    end subroutine read_int32

    subroutine read_images(unit, height, width, x)
        integer(int32), intent(in) :: unit, height, width
        real(real64), intent(out) :: x(:,:)
        integer(int8), allocatable :: buffer(:)
        integer(int32) :: i
        
        if(allocated(buffer)) deallocate(buffer)
        allocate(buffer(height * width))

        do i = 1, size(x,1)
            read(unit) buffer
            x(i,:) = real(buffer,real64)
            where (x(i,:) < 0.0_real64) x(i,:) = x(i,:) + 256.0_real64
            x(i,:) = x(i,:) / 255.0_real64
        end do
    end subroutine read_images

    subroutine read_label(unit,y)
        real(real64), intent(inout) :: y(:,:)
        integer(int32), intent(in) :: unit
        integer :: i
        integer(int8) :: label
        integer(int32) :: label_val

        do i = 1, size(y,1)
            read(unit) label 

            label_val = int(label, int32)
            if (label_val < 0) label_val = label_val + 256
            y(i,:) = 0.0_real64
            y(i, label_val + 1) = 1.0_real64
        end do
    end subroutine read_label
end module Readbin