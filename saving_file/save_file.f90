module SaveingData
    use, intrinsic :: iso_fortran_env
    use Network
    implicit none

    private
    public save_network, load_network, save_pytorch,load_pytorch
contains
    subroutine save_network(net,filename)
        class(layer), intent(in) :: net(:)
        character(len=*), intent(in) :: filename
        integer(int64) :: i

        open(unit=10, file=filename, form='unformatted', access='stream', status='new')
        write(10) int(size(net),int64)
        do i = 1, size(net)
            !b(current_neuron),W(prev_neuron,current_neuron)
           write(10) int(size(net(i)%b),int64), int(size(net(i)%W,1),int64), net(i)%W, net(i)%b
        end do
        close(10)

    end subroutine save_network

    subroutine load_network(net,filename,N)
        type(layer),allocatable, intent(out) :: net(:)
        character(len=*), intent(in) :: filename
        integer(int64), intent(out) :: N
        integer(int64) :: i,prev_neuron,current_neuron

        open(unit=10, file=filename, form='unformatted', access='stream', status='old')
        read(10) N

        if (allocated(net)) deallocate(net)
        allocate(net(N))

        do i = 1, N
            read(10) current_neuron, prev_neuron
            call intialize_layer(net(i), prev_neuron, current_neuron, 1_int64)
            read(10) net(i)%W, net(i)%b
        end do

        close(10)
    end subroutine

    subroutine save_pytorch(net,filename)
        class(layer), intent(in) :: net(:)
        character(len=*), intent(in) :: filename
        integer :: i, file_unit
        
        file_unit = 10
        open(unit=file_unit, file=filename, form='unformatted', access='stream', status='new')
        write(file_unit) int(size(net), int64)

        do i = 1, size(net)
            write(file_unit) int(size(net(i)%b),int64), int(size(net(i)%W,1),int64)

            write(file_unit) transpose(net(i)%W)
            write(file_unit) net(i)%b
        end do
        close(file_unit)
    end subroutine save_pytorch

    subroutine load_pytorch(net,filename,N)
        type(layer), allocatable ,intent(out) :: net(:)
        character(len=*), intent(in) :: filename
        integer(int64), intent(inout) :: N
        integer :: file_unit
        integer(int64) :: size_b,size_w,i
        real(real64), allocatable :: W(:,:)

        file_unit = 10
        open(unit=file_unit, file=filename, form='unformatted', access='stream', status='old')

        read(file_unit) N
        
        if (allocated(net)) deallocate(net)
        allocate(net(N))

        do i = 1, N
            read(file_unit) size_b, size_w
            call intialize_layer(net(i),size_w,size_b,1_int64)
            if (allocated(W)) deallocate(W)
            allocate(W(size_b, size_w))
            read(file_unit) W
            net(i)%W = transpose(W)
            read(file_unit) net(i)%b
        end do
        if (allocated(W)) deallocate(W)
        close(file_unit)
    end subroutine load_pytorch
end module SaveingData